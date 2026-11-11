package online.search;
public class ThreadPoolMonitor {
	public static long stuckCheckThreshold = Long.MAX_VALUE, stuckThreshold = Long.MAX_VALUE;
	public ThreadPoolMonitor(int n) {}
	public void setActive(int n) {}
	public boolean stuck() { return false; }
}
