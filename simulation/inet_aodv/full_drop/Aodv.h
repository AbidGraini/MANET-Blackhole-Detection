//
// Copyright (C) 2014 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//

#ifndef __INET_AODV_H
#define __INET_AODV_H

#include <map>
#include "inet/mobility/contract/IMobility.h"
#include "inet/power/storage/SimpleEpEnergyStorage.h"
#include "inet/common/ModuleRefByPar.h"
#include "inet/networklayer/contract/IInterfaceTable.h"
#include "inet/networklayer/contract/IL3AddressType.h"
#include "inet/networklayer/contract/INetfilter.h"
#include "inet/networklayer/contract/IRoutingTable.h"
#include "inet/routing/aodv/AodvControlPackets_m.h"
#include "inet/routing/aodv/AodvRouteData.h"
#include "inet/routing/base/RoutingProtocolBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/transportlayer/udp/UdpHeader_m.h"

namespace inet {
namespace aodv {


// AODV routing implementation with Blackhole attack support.
class INET_API Aodv : public RoutingProtocolBase, public NetfilterBase::HookBase, public UdpSocket::ICallback, public cListener
{
  protected:


    class INET_API RreqIdentifier {
      public:
        L3Address originatorAddr;
        unsigned int rreqID;
        RreqIdentifier(const L3Address& originatorAddr, unsigned int rreqID) : originatorAddr(originatorAddr), rreqID(rreqID) {};
        bool operator==(const RreqIdentifier& other) const
        {
            return this->originatorAddr == other.originatorAddr && this->rreqID == other.rreqID;
        }
    };

    class INET_API RreqIdentifierCompare {
      public:
        bool operator()(const RreqIdentifier& lhs, const RreqIdentifier& rhs) const
        {
            if (lhs.originatorAddr < rhs.originatorAddr)
                return true;
            else if (lhs.originatorAddr > rhs.originatorAddr)
                return false;
            else
                return lhs.rreqID < rhs.rreqID;
        }
    };


    const IL3AddressType *addressType = nullptr;


    simsignal_t rreqSentSignal;
    simsignal_t rrepSentSignal;
    simsignal_t rreqDropSignal;
    simsignal_t dataDropSignal;

    simsignal_t rrepSpoofedSignal;
    simsignal_t numPacketsForwardedSignal;
    simsignal_t numRREQsReceivedSignal;

    simsignal_t numRREQsSentSignal;
    simsignal_t numRREPsSentSignal;
    simsignal_t numRREQsForwardedSignal;


        cMessage *energyTimer = nullptr;


        inet::L3Address myAddress;
    bool isBlackHole = false;
    cModule *host = nullptr;
    ModuleRefByPar<IRoutingTable> routingTable;
    ModuleRefByPar<IInterfaceTable> interfaceTable;
    ModuleRefByPar<INetfilter> networkProtocol;
    UdpSocket socket;
    bool usingIpv6 = false;


    unsigned int rerrRatelimit = 0;
    unsigned int aodvUDPPort = 0;
    bool askGratuitousRREP = false;
    bool useHelloMessages = false;
    bool destinationOnlyFlag = false;
    simtime_t maxJitter;
    simtime_t activeRouteTimeout;
    simtime_t helloInterval;
    unsigned int netDiameter = 0;
    unsigned int rreqRetries = 0;
    unsigned int rreqRatelimit = 0;
    unsigned int timeoutBuffer = 0;
    unsigned int ttlStart = 0;
    unsigned int ttlIncrement = 0;
    unsigned int ttlThreshold = 0;
    unsigned int localAddTTL = 0;
    unsigned int allowedHelloLoss = 0;
    simtime_t nodeTraversalTime;
    cPar *jitterPar = nullptr;
    cPar *periodicJitter = nullptr;


    simtime_t deletePeriod;
    simtime_t myRouteTimeout;
    simtime_t blacklistTimeout;
    simtime_t netTraversalTime;
    simtime_t nextHopWait;
    simtime_t pathDiscoveryTime;


    unsigned int rreqId = 0;
    unsigned int sequenceNum = 0;
    std::map<L3Address, WaitForRrep *> waitForRREPTimers;
    std::map<RreqIdentifier, simtime_t, RreqIdentifierCompare> rreqsArrivalTime;
    L3Address failedNextHop;
    std::map<L3Address, simtime_t> blacklist;
    unsigned int rerrCount = 0;
    unsigned int rreqCount = 0;
    simtime_t lastBroadcastTime;
    std::map<L3Address, unsigned int> addressToRreqRetries;


    cMessage *helloMsgTimer = nullptr;
    cMessage *expungeTimer = nullptr;
    cMessage *counterTimer = nullptr;
    cMessage *rrepAckTimer = nullptr;
    cMessage *blacklistTimer = nullptr;


    simtime_t rebootTime;


    std::multimap<L3Address, Packet *> targetAddressToDelayedPackets;

  protected:
    void handleMessageWhenUp(cMessage *msg) override;
    void initialize(int stage) override;
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }


    void startRouteDiscovery(const L3Address& target, unsigned int timeToLive = 0);
    void completeRouteDiscovery(const L3Address& target);
    bool hasOngoingRouteDiscovery(const L3Address& destAddr);
    void cancelRouteDiscovery(const L3Address& destAddr);


    void updateRoutingTable(IRoute *route, const L3Address& nextHop, unsigned int hopCount, bool hasValidDestNum, unsigned int destSeqNum, bool isActive, simtime_t lifeTime);
    IRoute *createRoute(const L3Address& destAddr, const L3Address& nextHop, unsigned int hopCount, bool hasValidDestNum, unsigned int destSeqNum, bool isActive, simtime_t lifeTime);
    bool updateValidRouteLifeTime(const L3Address& destAddr, simtime_t lifetime);
    void scheduleExpungeRoutes();
    void expungeRoutes();


    const Ptr<RrepAck> createRREPACK();
    const Ptr<Rrep> createHelloMessage();
    const Ptr<Rreq> createRREQ(const L3Address& destAddr);
    const Ptr<Rrep> createRREP(const Ptr<Rreq>& rreq, IRoute *destRoute, IRoute *originatorRoute, const L3Address& sourceAddr);
    const Ptr<Rrep> createGratuitousRREP(const Ptr<Rreq>& rreq, IRoute *originatorRoute);
    const Ptr<Rerr> createRERR(const std::vector<UnreachableNode>& unreachableNodes);


    void handleRREP(const Ptr<Rrep>& rrep, const L3Address& sourceAddr);
    void handleRREQ(const Ptr<Rreq>& rreq, const L3Address& sourceAddr, unsigned int timeToLive);
    void handleRERR(const Ptr<const Rerr>& rerr, const L3Address& sourceAddr);
    void handleHelloMessage(const Ptr<Rrep>& helloMessage);
    void handleRREPACK(const Ptr<const RrepAck>& rrepACK, const L3Address& neighborAddr);


    void sendRREQ(const Ptr<Rreq>& rreq, const L3Address& destAddr, unsigned int timeToLive);
    void sendRREPACK(const Ptr<RrepAck>& rrepACK, const L3Address& destAddr);
    void sendRREP(const Ptr<Rrep>& rrep, const L3Address& destAddr, unsigned int timeToLive);
    void sendGRREP(const Ptr<Rrep>& grrep, const L3Address& destAddr, unsigned int timeToLive);


    void forwardRREP(const Ptr<Rrep>& rrep, const L3Address& destAddr, unsigned int timeToLive);
    void forwardRREQ(const Ptr<Rreq>& rreq, unsigned int timeToLive);


    void handleRREPACKTimer();
    void handleBlackListTimer();
    void sendHelloMessagesIfNeeded();
    void handleWaitForRREP(WaitForRrep *rrepTimer);


    void sendRERRWhenNoRouteToForward(const L3Address& unreachableAddr);
    void handleLinkBreakSendRERR(const L3Address& unreachableAddr);
    virtual void receiveSignal(cComponent *source, simsignal_t signalID, cObject *obj, cObject *details) override;


    Result ensureRouteForDatagram(Packet *datagram);
    virtual Result datagramPreRoutingHook(Packet *datagram) override { Enter_Method("datagramPreRoutingHook"); return ensureRouteForDatagram(datagram); }
    virtual Result datagramForwardHook(Packet *datagram) override;
    virtual Result datagramPostRoutingHook(Packet *datagram) override { return ACCEPT; }
    virtual Result datagramLocalInHook(Packet *datagram) override { return ACCEPT; }
    virtual Result datagramLocalOutHook(Packet *datagram) override { Enter_Method("datagramLocalOutHook"); return ensureRouteForDatagram(datagram); }
    void delayDatagram(Packet *datagram);


    L3Address getSelfIPAddress() const;
    void sendAODVPacket(const Ptr<AodvControlPacket>& packet, const L3Address& destAddr, unsigned int timeToLive, double delay);
    void processPacket(Packet *pk);
    void clearState();
    void checkIpVersionAndPacketTypeCompatibility(AodvControlPacketType packetType);


    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;


    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

  public:

    Aodv();
    virtual ~Aodv();
};

}
}

#endif

