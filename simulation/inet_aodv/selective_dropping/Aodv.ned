//
// Copyright (C) 2014 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//

package inet.routing.aodv;

import inet.applications.contract.IApp;


// AODV routing protocol module (RFC 3561).
simple Aodv like IApp
{
    parameters:
        // Enables Blackhole behavior for this node.
        bool isblackhole = default(false);

        // Probability of dropping an intercepted transit packet.
        double blackholeDropProbability = default(1.0);
        @class("aodv::Aodv");
        @display("i=block/network2");
        string routingTableModule = default("^.ipv4.routingTable");
        string interfaceTableModule = default("^.interfaceTable");
        string networkProtocolModule = default("^.ipv4.ip");
        bool askGratuitousRREP = default(false);
        bool useHelloMessages = default(false);
        bool useLocalRepair = default(false);
        bool destinationOnlyFlag = default(false);
        int udpPort = default(654);
        string interface = default("wlan0");

        double maxPeriodicJitter @unit(s) = default(helloInterval / 4);
        volatile double periodicJitter @unit(s) = default(uniform(0s, maxPeriodicJitter));


        double maxJitter @unit(s) = default(5ms);
        volatile double jitter @unit(s) = default(uniform(0ms, maxJitter));

        double helloInterval @unit(s) = default(1s);
        int allowedHelloLoss = default(2);
        double activeRouteTimeout @unit(s) = default(3s);


        int netDiameter = default(35);
        double nodeTraversalTime @unit(s) = default(0.04s);
        int rerrRatelimit = default(10);
        int rreqRetries = default(2);
        int rreqRatelimit = default(10);
        int timeoutBuffer = default(2);
        int ttlStart = default(2);
        int ttlIncrement = default(2);
        int ttlThreshold = default(7);
        int localAddTTL = default(2);

        double myRouteTimeout @unit(s) = default(2 * activeRouteTimeout);
        double deletePeriod @unit(s) = default(5 * max(activeRouteTimeout, helloInterval));
        double blacklistTimeout @unit(s) = default(rreqRetries * netTraversalTime);
        double netTraversalTime @unit(s) = default(2 * nodeTraversalTime * netDiameter);
        double nextHopWait @unit(s) = default(nodeTraversalTime + 0.01s);
        double pathDiscoveryTime @unit(s) = default(2 * netTraversalTime);
        @signal[rreqDrop](type="long");
        @statistic[rreqDropCount](source="rreqDrop"; record=count;);
        @signal[dataDrop](type="long");
        @statistic[dataDropCount](source="dataDrop"; record=count;);

         @signal[rrepSpoofed](type="long");
        @statistic[rrepSpoofedCount](source="rrepSpoofed"; record=count;);

        @signal[numPacketsForwarded](type="long");
        @statistic[numPacketsForwarded](source="numPacketsForwarded"; record=count;);
        @signal[numRREQsReceived](type="long");
        @statistic[numRREQsReceived](source="numRREQsReceived"; record=count;);
        @signal[numRREQsSent](type="long");
       @statistic[numRREQsSent](source="numRREQsSent"; record=count);
       @signal[numRREPsSent](type="long");
       @statistic[numRREPsSent](source="numRREPsSent"; record=count);
       @signal[numRREQsForwarded](type="long");
       @statistic[numRREQsForwarded](source="numRREQsForwarded"; record=count);

    gates:
        input socketIn;
        output socketOut;
}

